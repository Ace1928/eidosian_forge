from ._base import *
class Nginx:

    @classmethod
    def loads(cls, data, conf=True):
        """
        Load an nginx configuration from a provided string.
        :param str data: nginx configuration
        :param bool conf: Load object(s) into a NginxConfig object?
        """
        f = NginxConfig() if conf else []
        lopen = []
        index = 0
        while True:
            m = re.compile('^\\s*events\\s*{').search(data[index:])
            if m:
                logger.debug('Open (Events)')
                e = Events()
                lopen.insert(0, e)
                index += m.end()
                continue
            m = re.compile('^\\s*http\\s*{').search(data[index:])
            if m:
                logger.debug('Open (Http)')
                h = Http()
                lopen.insert(0, h)
                index += m.end()
                continue
            m = re.compile('^\\s*stream\\s*{').search(data[index:])
            if m:
                logger.debug('Open (Stream)')
                s = Stream()
                lopen.insert(0, s)
                index += m.end()
                continue
            m = re.compile('^\\s*server\\s*{').search(data[index:])
            if m:
                logger.debug('Open (Server)')
                s = Server()
                lopen.insert(0, s)
                index += m.end()
                continue
            n = re.compile('(?!\\B"[^"]*);(?![^"]*"\\B)')
            m = re.compile('^\\s*location\\s+(.*?".*?".*?|.*?)\\s*{').search(data[index:])
            if m and (not n.search(m.group())):
                logger.debug('Open (Location) {0}'.format(m.group(1)))
                l = Location(m.group(1))
                lopen.insert(0, l)
                index += m.end()
                continue
            m = re.compile('^\\s*if\\s+(.*?".*?".*?|.*?)\\s*{').search(data[index:])
            if m and (not n.search(m.group())):
                logger.debug('Open (If) {0}'.format(m.group(1)))
                ifs = If(m.group(1))
                lopen.insert(0, ifs)
                index += m.end()
                continue
            m = re.compile('^\\s*upstream\\s+(.*?)\\s*{').search(data[index:])
            if m and (not n.search(m.group())):
                logger.debug('Open (Upstream) {0}'.format(m.group(1)))
                u = Upstream(m.group(1))
                lopen.insert(0, u)
                index += m.end()
                continue
            m = re.compile('^\\s*geo\\s+(.*?".*?".*?|.*?)\\s*{').search(data[index:])
            if m and (not n.search(m.group())):
                logger.debug('Open (Geo) {0}'.format(m.group(1)))
                g = Geo(m.group(1))
                lopen.insert(0, g)
                index += m.end()
                continue
            m = re.compile('^\\s*map\\s+(.*?".*?".*?|.*?)\\s*{').search(data[index:])
            if m and (not n.search(m.group())):
                logger.debug('Open (Map) {0}'.format(m.group(1)))
                g = Map(m.group(1))
                lopen.insert(0, g)
                index += m.end()
                continue
            m = re.compile('^\\s*limit_except\\s+(.*?".*?".*?|.*?)\\s*{').search(data[index:])
            if m and (not n.search(m.group())):
                logger.debug('Open (LimitExcept) {0}'.format(m.group(1)))
                l = LimitExcept(m.group(1))
                lopen.insert(0, l)
                index += m.end()
                continue
            m = re.compile('^\\s*types\\s*{').search(data[index:])
            if m:
                logger.debug('Open (Types)')
                l = Types()
                lopen.insert(0, l)
                index += m.end()
                continue
            m = re.compile('^(\\s*)#[ \\r\\t\\f]*(.*?)\\n').search(data[index:])
            if m:
                logger.debug('Comment ({0})'.format(m.group(2)))
                c = Comment(m.group(2), inline='\n' not in m.group(1))
                if lopen and isinstance(lopen[0], Container):
                    lopen[0].add(c)
                else:
                    f.add(c) if conf else f.append(c)
                index += m.end() - 1
                continue
            m = re.compile('^\\s*}').search(data[index:])
            if m:
                if isinstance(lopen[0], Container):
                    logger.debug('Close ({0})'.format(lopen[0].__class__.__name__))
                    c = lopen[0]
                    lopen.pop(0)
                    if lopen and isinstance(lopen[0], Container):
                        lopen[0].add(c)
                    else:
                        f.add(c) if conf else f.append(c)
                index += m.end()
                continue
            if ';' not in data[index:] and '}' in data[index:]:
                raise ParseError("Config syntax, missing ';' at index: {}".format(index))
            double = '\\s*"[^"]*"'
            single = "\\s*\\'[^\\']*\\'"
            normal = '\\s*[^;\\s]*'
            s1 = '{}|{}|{}'.format(double, single, normal)
            s = '^\\s*({})\\s*((?:{})+);'.format(s1, s1)
            m = re.compile(s).search(data[index:])
            if m:
                logger.debug('Key {0} {1}'.format(m.group(1), m.group(2)))
                k = Key(m.group(1), m.group(2))
                if lopen and isinstance(lopen[0], (Container, Server)):
                    lopen[0].add(k)
                else:
                    f.add(k) if conf else f.append(k)
                index += m.end()
                continue
            m = re.compile('^\\s*(\\S+);').search(data[index:])
            if m:
                logger.debug('Key {0}'.format(m.group(1)))
                k = Key(m.group(1), '')
                if lopen and isinstance(lopen[0], (Container, Server)):
                    lopen[0].add(k)
                else:
                    f.add(k) if conf else f.append(k)
                index += m.end()
                continue
            break
        return f

    @classmethod
    def load(cls, pathlike_or_filestr):
        """
        Load an nginx configuration from a provided file-like object.
        :param obj fobj: nginx configuration
        """
        data = PathIO(pathlike_or_filestr).read_text() if isinstance(pathlike_or_filestr, str) else pathlike_or_filestr.read()
        return Nginx.loads(data)

    @classmethod
    def dumps(cls, obj):
        """
        Dump an nginx configuration to a string.
        :param obj obj: nginx object (NginxConfig, Server, Container)
        :returns: nginx configuration as string
        """
        return ''.join(obj.as_strings)

    @classmethod
    def dump(cls, obj, pathlike_or_filestr):
        """
        Write an nginx configuration to a file-like object.
        :param obj obj: nginx object (NginxConfig, Server, Container)
        :param obj fobj: file-like object to write to
        :returns: file-like object that was written to
        """
        pathlike = PathIO(pathlike_or_filestr) if isinstance(pathlike_or_filestr, str) else pathlike_or_filestr
        pathlike.write(Nginx.dumps(obj))
        return pathlike