class SH_FLAGS(object):
    """ Flag values for the sh_flags field of section headers
    """
    SHF_WRITE = 1
    SHF_ALLOC = 2
    SHF_EXECINSTR = 4
    SHF_MERGE = 16
    SHF_STRINGS = 32
    SHF_INFO_LINK = 64
    SHF_LINK_ORDER = 128
    SHF_OS_NONCONFORMING = 256
    SHF_GROUP = 512
    SHF_TLS = 1024
    SHF_COMPRESSED = 2048
    SHF_MASKOS = 267386880
    SHF_EXCLUDE = 2147483648
    SHF_MASKPROC = 4026531840