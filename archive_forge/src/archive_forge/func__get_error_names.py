import re
def _get_error_names(mod, type_, code):
    t_name = _get_value_name(mod, type_, 'OFPET_')
    if t_name == 'Unknown':
        return ('Unknown', 'Unknown')
    if t_name == 'OFPET_FLOW_MONITOR_FAILED':
        c_name_p = 'OFPMOFC_'
    else:
        c_name_p = 'OFP'
        for m in re.findall('_(.)', t_name):
            c_name_p += m.upper()
        c_name_p += 'C_'
    c_name = _get_value_name(mod, code, c_name_p)
    return (t_name, c_name)