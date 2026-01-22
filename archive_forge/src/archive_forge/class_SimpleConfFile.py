from typing import Optional
class SimpleConfFile:
    """
    Simple configuration file parser superclass.

    Filters out comments and empty lines (which includes lines that only
    contain comments).

    To use this class, override parseLine or parseFields.
    """
    commentChar = '#'
    defaultFilename: Optional[str] = None

    def parseFile(self, file=None):
        """
        Parse a configuration file

        If file is None and self.defaultFilename is set, it will open
        defaultFilename and use it.
        """
        close = False
        if file is None and self.defaultFilename:
            file = open(self.defaultFilename)
            close = True
        try:
            for line in file.readlines():
                comment = line.find(self.commentChar)
                if comment != -1:
                    line = line[:comment]
                line = line.strip()
                if not line:
                    continue
                self.parseLine(line)
        finally:
            if close:
                file.close()

    def parseLine(self, line):
        """
        Override this.

        By default, this will split the line on whitespace and call
        self.parseFields (catching any errors).
        """
        try:
            self.parseFields(*line.split())
        except ValueError:
            raise InvalidInetdConfError('Invalid line: ' + repr(line))

    def parseFields(self, *fields):
        """
        Override this.
        """