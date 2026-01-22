from email.utils import parseaddr
def _parse_id(self, id):
    if id.find(b'<') == -1:
        return (id, b'')
    else:
        return parseaddr(id)