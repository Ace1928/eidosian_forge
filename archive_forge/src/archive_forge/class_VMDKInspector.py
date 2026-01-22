import struct
from oslo_log import log as logging
class VMDKInspector(FileInspector):
    """vmware VMDK format (monolithicSparse and streamOptimized variants only)

    This needs to store the 512 byte header and the descriptor region
    which should be just after that. The descriptor region is some
    variable number of 512 byte sectors, but is just text defining the
    layout of the disk.
    """
    DESC_OFFSET = 512
    DESC_MAX_SIZE = (1 << 20) - 1

    def __init__(self, *a, **k):
        super(VMDKInspector, self).__init__(*a, **k)
        self.new_region('header', CaptureRegion(0, 512))

    def post_process(self):
        if not self.region('header').complete:
            return
        sig, ver, _flags, _sectors, _grain, desc_sec, desc_num = struct.unpack('<4sIIQQQQ', self.region('header').data[:44])
        if sig != b'KDMV':
            raise ImageFormatError('Signature KDMV not found: %r' % sig)
        if ver not in (1, 2, 3):
            raise ImageFormatError('Unsupported format version %i' % ver)
        desc_offset = desc_sec * 512
        desc_size = min(desc_num * 512, self.DESC_MAX_SIZE)
        if desc_offset != self.DESC_OFFSET:
            raise ImageFormatError('Wrong descriptor location')
        if not self.has_region('descriptor'):
            self.new_region('descriptor', CaptureRegion(desc_offset, desc_size))

    @property
    def format_match(self):
        return self.region('header').data.startswith(b'KDMV')

    @property
    def virtual_size(self):
        if not self.has_region('descriptor'):
            return 0
        descriptor_rgn = self.region('descriptor')
        if not descriptor_rgn.complete:
            return 0
        descriptor = descriptor_rgn.data
        type_idx = descriptor.index(b'createType="') + len(b'createType="')
        type_end = descriptor.find(b'"', type_idx)
        if type_end - type_idx < 64:
            vmdktype = descriptor[type_idx:type_end]
        else:
            vmdktype = b'formatnotfound'
        if vmdktype not in (b'monolithicSparse', b'streamOptimized'):
            LOG.warning('Unsupported VMDK format %s', vmdktype)
            return 0
        _sig, _ver, _flags, sectors, _grain, _desc_sec, _desc_num = struct.unpack('<IIIQQQQ', self.region('header').data[:44])
        return sectors * 512

    def __str__(self):
        return 'vmdk'