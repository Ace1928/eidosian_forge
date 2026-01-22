from defusedxml.lxml import parse
def _get_vmdk_name_from_ovf(root):
    ns_ovf = '{{{0}}}'.format(root.nsmap['ovf'])
    disk = root.find('./{0}DiskSection/{0}Disk'.format(ns_ovf))
    file_id = disk.get('{0}fileRef'.format(ns_ovf))
    f = root.find('./{0}References/{0}File[@{0}id="{1}"]'.format(ns_ovf, file_id))
    return f.get('{0}href'.format(ns_ovf))