import io
@classmethod
def _format_family_header(cls):
    ret = ''
    ret += 'Status codes: * valid, > best\n'
    ret += 'Origin codes: i - IGP, e - EGP, ? - incomplete\n'
    ret += cls.fmtstr.format('', 'Network', 'Labels', 'Next Hop', 'Reason', 'Metric', 'LocPrf', 'Path')
    return ret