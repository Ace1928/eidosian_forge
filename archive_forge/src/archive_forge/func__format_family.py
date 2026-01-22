import io
@classmethod
def _format_family(cls, dest_list):
    msg = io.StringIO()

    def _append_path_info(buff, path, is_best, show_prefix):
        aspath = path.get('aspath')
        origin = path.get('origin')
        if origin:
            aspath.append(origin)
        bpr = path.get('bpr')
        next_hop = path.get('nexthop')
        med = path.get('metric')
        labels = path.get('labels')
        localpref = path.get('localpref')
        path_status = '*'
        if is_best:
            path_status += '>'
        prefix = ''
        if show_prefix:
            prefix = path.get('prefix')
        buff.write(cls.fmtstr.format(path_status, prefix, str(labels), str(next_hop), bpr, str(med), str(localpref), ' '.join(map(str, aspath))))
    for dist in dest_list:
        for idx, path in enumerate(dist.get('paths')):
            _append_path_info(msg, path, path['best'], idx == 0)
    ret = msg.getvalue()
    msg.close()
    return ret