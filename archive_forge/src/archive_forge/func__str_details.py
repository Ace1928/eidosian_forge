@property
def _str_details(self):
    return '\n    ' + '\n    '.join((x._str_details.strip() if isinstance(x, _TargetInvalid) else str(x) for x in self.exceptions))