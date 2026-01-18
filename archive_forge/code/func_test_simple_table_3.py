from numpy.testing import assert_equal
from statsmodels.iolib.table import SimpleTable, default_txt_fmt
from statsmodels.iolib.table import default_latex_fmt
from statsmodels.iolib.table import default_html_fmt
import pandas
from statsmodels.regression.linear_model import OLS
def test_simple_table_3(self):
    desired = '\n==============================\n           header s1 header d1\n------------------------------\nstub R1 C1  10.30312  10.73999\nstub R2 C1  90.30312  90.73999\n           header s2 header d2\n------------------------------\nstub R1 C2  50.95038  50.65765\nstub R2 C2  40.95038  40.65765\n------------------------------\n'
    data1 = [[10.30312, 10.73999], [90.30312, 90.73999]]
    data2 = [[50.95038, 50.65765], [40.95038, 40.65765]]
    stubs1 = ['stub R1 C1', 'stub R2 C1']
    stubs2 = ['stub R1 C2', 'stub R2 C2']
    header1 = ['header s1', 'header d1']
    header2 = ['header s2', 'header d2']
    actual1 = SimpleTable(data1, header1, stubs1, txt_fmt=default_txt_fmt)
    actual2 = SimpleTable(data2, header2, stubs2, txt_fmt=default_txt_fmt)
    actual1.extend(actual2)
    actual = '\n%s\n' % actual1.as_text()
    assert_equal(desired, str(actual))