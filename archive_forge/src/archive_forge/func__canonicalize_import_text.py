from .errors import BzrError, InternalBzrError
def _canonicalize_import_text(self, text):
    """Take a list of imports, and split it into regularized form.

        This is meant to take regular import text, and convert it to
        the forms that the rest of the converters prefer.
        """
    out = []
    cur = None
    for line in text.split('\n'):
        line = line.strip()
        loc = line.find('#')
        if loc != -1:
            line = line[:loc].strip()
        if not line:
            continue
        if cur is not None:
            if line.endswith(')'):
                out.append(cur + ' ' + line[:-1])
                cur = None
            else:
                cur += ' ' + line
        elif '(' in line and ')' not in line:
            cur = line.replace('(', '')
        else:
            out.append(line.replace('(', '').replace(')', ''))
    if cur is not None:
        raise InvalidImportLine(cur, 'Unmatched parenthesis')
    return out