from email.utils import parseaddr
class UserMapper:

    def __init__(self, lines):
        """Create a user-mapper from a list of lines.

        Blank lines and comment lines (starting with #) are ignored.
        Otherwise lines are of the form:

          old-id = new-id

        Each id may be in the following forms:

          name <email>
          name

        If old-id has the value '@', then new-id is the domain to use
        when generating an email from a user-id.
        """
        self._parse(lines)

    def _parse(self, lines):
        self._user_map = {}
        self._default_domain = None
        for line in lines:
            line = line.strip()
            if len(line) == 0 or line.startswith(b'#'):
                continue
            old, new = line.split(b'=', 1)
            old = old.strip()
            new = new.strip()
            if old == b'@':
                self._default_domain = new
                continue
            old_name, old_email = self._parse_id(old)
            new_name, new_email = self._parse_id(new)
            self._user_map[old_name, old_email] = (new_name, new_email)

    def _parse_id(self, id):
        if id.find(b'<') == -1:
            return (id, b'')
        else:
            return parseaddr(id)

    def map_name_and_email(self, name, email):
        """Map a name and an email to the preferred name and email.

        :param name: the current name
        :param email: the current email
        :result: the preferred name and email
        """
        try:
            new_name, new_email = self._user_map[name, email]
        except KeyError:
            new_name = name
            if self._default_domain and (not email):
                new_email = b'%s@%s' % (name, self._default_domain)
            else:
                new_email = email
        return (new_name, new_email)