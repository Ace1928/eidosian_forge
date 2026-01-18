from numpy.testing import assert_equal
from statsmodels.iolib.table import SimpleTable, default_txt_fmt
from statsmodels.iolib.table import default_latex_fmt
from statsmodels.iolib.table import default_html_fmt
import pandas
from statsmodels.regression.linear_model import OLS
def test_txt_fmt1(self):
    desired = '\n*****************************\n*       * header1 * header2 *\n*****************************\n* stub1 *    0.00 *       1 *\n* stub2 *    2.00 *       3 *\n*****************************\n'
    actual = '\n%s\n' % tbl.as_text()
    assert_equal(actual, desired)