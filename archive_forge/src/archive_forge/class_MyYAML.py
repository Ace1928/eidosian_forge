import sys
import textwrap
from pathlib import Path
class MyYAML(srsly.ruamel_yaml.YAML):
    """auto dedent string parameters on load"""

    def load(self, stream):
        if isinstance(stream, str):
            if stream and stream[0] == '\n':
                stream = stream[1:]
            stream = textwrap.dedent(stream)
        return srsly.ruamel_yaml.YAML.load(self, stream)

    def load_all(self, stream):
        if isinstance(stream, str):
            if stream and stream[0] == '\n':
                stream = stream[1:]
            stream = textwrap.dedent(stream)
        for d in srsly.ruamel_yaml.YAML.load_all(self, stream):
            yield d

    def dump(self, data, **kw):
        from srsly.ruamel_yaml.compat import StringIO, BytesIO
        assert ('stream' in kw) ^ ('compare' in kw)
        if 'stream' in kw:
            return srsly.ruamel_yaml.YAML.dump(data, **kw)
        lkw = kw.copy()
        expected = textwrap.dedent(lkw.pop('compare'))
        unordered_lines = lkw.pop('unordered_lines', False)
        if expected and expected[0] == '\n':
            expected = expected[1:]
        lkw['stream'] = st = StringIO()
        srsly.ruamel_yaml.YAML.dump(self, data, **lkw)
        res = st.getvalue()
        print(res)
        if unordered_lines:
            res = sorted(res.splitlines())
            expected = sorted(expected.splitlines())
        assert res == expected

    def round_trip(self, stream, **kw):
        from srsly.ruamel_yaml.compat import StringIO, BytesIO
        assert isinstance(stream, (srsly.ruamel_yaml.compat.text_type, str))
        lkw = kw.copy()
        if stream and stream[0] == '\n':
            stream = stream[1:]
        stream = textwrap.dedent(stream)
        data = srsly.ruamel_yaml.YAML.load(self, stream)
        outp = lkw.pop('outp', stream)
        lkw['stream'] = st = StringIO()
        srsly.ruamel_yaml.YAML.dump(self, data, **lkw)
        res = st.getvalue()
        if res != outp:
            diff(outp, res, 'input string')
        assert res == outp

    def round_trip_all(self, stream, **kw):
        from srsly.ruamel_yaml.compat import StringIO, BytesIO
        assert isinstance(stream, (srsly.ruamel_yaml.compat.text_type, str))
        lkw = kw.copy()
        if stream and stream[0] == '\n':
            stream = stream[1:]
        stream = textwrap.dedent(stream)
        data = list(srsly.ruamel_yaml.YAML.load_all(self, stream))
        outp = lkw.pop('outp', stream)
        lkw['stream'] = st = StringIO()
        srsly.ruamel_yaml.YAML.dump_all(self, data, **lkw)
        res = st.getvalue()
        if res != outp:
            diff(outp, res, 'input string')
        assert res == outp