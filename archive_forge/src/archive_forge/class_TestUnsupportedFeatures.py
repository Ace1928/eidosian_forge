from io import StringIO
import os
from pathlib import Path
import pytest
from pandas.errors import ParserError
import pandas._testing as tm
from pandas.io.parsers import read_csv
import pandas.io.parsers.readers as parsers
class TestUnsupportedFeatures:

    def test_mangle_dupe_cols_false(self):
        data = 'a b c\n1 2 3'
        for engine in ('c', 'python'):
            with pytest.raises(TypeError, match='unexpected keyword'):
                read_csv(StringIO(data), engine=engine, mangle_dupe_cols=True)

    def test_c_engine(self):
        data = 'a b c\n1 2 3'
        msg = 'does not support'
        depr_msg = "The 'delim_whitespace' keyword in pd.read_csv is deprecated"
        with pytest.raises(ValueError, match=msg):
            with tm.assert_produces_warning(FutureWarning, match=depr_msg):
                read_csv(StringIO(data), engine='c', sep=None, delim_whitespace=False)
        with pytest.raises(ValueError, match=msg):
            read_csv(StringIO(data), engine='c', sep='\\s')
        with pytest.raises(ValueError, match=msg):
            read_csv(StringIO(data), engine='c', sep='\t', quotechar=chr(128))
        with pytest.raises(ValueError, match=msg):
            read_csv(StringIO(data), engine='c', skipfooter=1)
        with tm.assert_produces_warning((parsers.ParserWarning, FutureWarning)):
            read_csv(StringIO(data), sep=None, delim_whitespace=False)
        with tm.assert_produces_warning(parsers.ParserWarning):
            read_csv(StringIO(data), sep='\\s')
        with tm.assert_produces_warning(parsers.ParserWarning):
            read_csv(StringIO(data), sep='\t', quotechar=chr(128))
        with tm.assert_produces_warning(parsers.ParserWarning):
            read_csv(StringIO(data), skipfooter=1)
        text = '                      A       B       C       D        E\none two three   four\na   b   10.0032 5    -0.5109 -2.3358 -0.4645  0.05076  0.3640\na   q   20      4     0.4473  1.4152  0.2834  1.00661  0.1744\nx   q   30      3    -0.6662 -0.5243 -0.3580  0.89145  2.5838'
        msg = 'Error tokenizing data'
        with pytest.raises(ParserError, match=msg):
            read_csv(StringIO(text), sep='\\s+')
        with pytest.raises(ParserError, match=msg):
            read_csv(StringIO(text), engine='c', sep='\\s+')
        msg = 'Only length-1 thousands markers supported'
        data = 'A|B|C\n1|2,334|5\n10|13|10.\n'
        with pytest.raises(ValueError, match=msg):
            read_csv(StringIO(data), thousands=',,')
        with pytest.raises(ValueError, match=msg):
            read_csv(StringIO(data), thousands='')
        msg = 'Only length-1 line terminators supported'
        data = 'a,b,c~~1,2,3~~4,5,6'
        with pytest.raises(ValueError, match=msg):
            read_csv(StringIO(data), lineterminator='~~')

    def test_python_engine(self, python_engine):
        from pandas.io.parsers.readers import _python_unsupported as py_unsupported
        data = '1,2,3,,\n1,2,3,4,\n1,2,3,4,5\n1,2,,,\n1,2,3,4,'
        for default in py_unsupported:
            msg = f'The {repr(default)} option is not supported with the {repr(python_engine)} engine'
            kwargs = {default: object()}
            with pytest.raises(ValueError, match=msg):
                read_csv(StringIO(data), engine=python_engine, **kwargs)

    def test_python_engine_file_no_iter(self, python_engine):

        class NoNextBuffer:

            def __init__(self, csv_data) -> None:
                self.data = csv_data

            def __next__(self):
                return self.data.__next__()

            def read(self):
                return self.data

            def readline(self):
                return self.data
        data = 'a\n1'
        msg = "'NoNextBuffer' object is not iterable|argument 1 must be an iterator"
        with pytest.raises(TypeError, match=msg):
            read_csv(NoNextBuffer(data), engine=python_engine)

    def test_pyarrow_engine(self):
        from pandas.io.parsers.readers import _pyarrow_unsupported as pa_unsupported
        data = '1,2,3,,\n        1,2,3,4,\n        1,2,3,4,5\n        1,2,,,\n        1,2,3,4,'
        for default in pa_unsupported:
            msg = f"The {repr(default)} option is not supported with the 'pyarrow' engine"
            kwargs = {default: object()}
            default_needs_bool = {'warn_bad_lines', 'error_bad_lines'}
            if default == 'dialect':
                kwargs[default] = 'excel'
            elif default in default_needs_bool:
                kwargs[default] = True
            elif default == 'on_bad_lines':
                kwargs[default] = 'warn'
            warn = None
            depr_msg = None
            if 'delim_whitespace' in kwargs:
                depr_msg = "The 'delim_whitespace' keyword in pd.read_csv is deprecated"
                warn = FutureWarning
            if 'verbose' in kwargs:
                depr_msg = "The 'verbose' keyword in pd.read_csv is deprecated"
                warn = FutureWarning
            with pytest.raises(ValueError, match=msg):
                with tm.assert_produces_warning(warn, match=depr_msg):
                    read_csv(StringIO(data), engine='pyarrow', **kwargs)

    def test_on_bad_lines_callable_python_or_pyarrow(self, all_parsers):
        sio = StringIO('a,b\n1,2')
        bad_lines_func = lambda x: x
        parser = all_parsers
        if all_parsers.engine not in ['python', 'pyarrow']:
            msg = "on_bad_line can only be a callable function if engine='python' or 'pyarrow'"
            with pytest.raises(ValueError, match=msg):
                parser.read_csv(sio, on_bad_lines=bad_lines_func)
        else:
            parser.read_csv(sio, on_bad_lines=bad_lines_func)