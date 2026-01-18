import numpy as np
import pytest
from ..rstutils import rst_table
def test_rst_table():
    R, C = (3, 4)
    cell_values = np.arange(R * C).reshape((R, C))
    assert rst_table(cell_values) == '+--------+--------+--------+--------+--------+\n|        | col[0] | col[1] | col[2] | col[3] |\n+========+========+========+========+========+\n| row[0] |  0.00  |  1.00  |  2.00  |  3.00  |\n| row[1] |  4.00  |  5.00  |  6.00  |  7.00  |\n| row[2] |  8.00  |  9.00  | 10.00  | 11.00  |\n+--------+--------+--------+--------+--------+'
    assert rst_table(cell_values, ['a', 'b', 'c']) == '+---+--------+--------+--------+--------+\n|   | col[0] | col[1] | col[2] | col[3] |\n+===+========+========+========+========+\n| a |  0.00  |  1.00  |  2.00  |  3.00  |\n| b |  4.00  |  5.00  |  6.00  |  7.00  |\n| c |  8.00  |  9.00  | 10.00  | 11.00  |\n+---+--------+--------+--------+--------+'
    with pytest.raises(ValueError):
        rst_table(cell_values, ['a', 'b'])
    with pytest.raises(ValueError):
        rst_table(cell_values, ['a', 'b', 'c', 'd'])
    assert rst_table(cell_values, None, ['1', '2', '3', '4']) == '+--------+-------+-------+-------+-------+\n|        |   1   |   2   |   3   |   4   |\n+========+=======+=======+=======+=======+\n| row[0] |  0.00 |  1.00 |  2.00 |  3.00 |\n| row[1] |  4.00 |  5.00 |  6.00 |  7.00 |\n| row[2] |  8.00 |  9.00 | 10.00 | 11.00 |\n+--------+-------+-------+-------+-------+'
    with pytest.raises(ValueError):
        rst_table(cell_values, None, ['1', '2', '3'])
    with pytest.raises(ValueError):
        rst_table(cell_values, None, list('12345'))
    assert rst_table(cell_values, title='A title') == '*******\nA title\n*******\n\n+--------+--------+--------+--------+--------+\n|        | col[0] | col[1] | col[2] | col[3] |\n+========+========+========+========+========+\n| row[0] |  0.00  |  1.00  |  2.00  |  3.00  |\n| row[1] |  4.00  |  5.00  |  6.00  |  7.00  |\n| row[2] |  8.00  |  9.00  | 10.00  | 11.00  |\n+--------+--------+--------+--------+--------+'
    assert rst_table(cell_values, val_fmt='{0}') == '+--------+--------+--------+--------+--------+\n|        | col[0] | col[1] | col[2] | col[3] |\n+========+========+========+========+========+\n| row[0] | 0      | 1      | 2      | 3      |\n| row[1] | 4      | 5      | 6      | 7      |\n| row[2] | 8      | 9      | 10     | 11     |\n+--------+--------+--------+--------+--------+'
    cell_values_back = np.arange(R * C)[::-1].reshape((R, C))
    cell_3d = np.dstack((cell_values, cell_values_back))
    assert rst_table(cell_3d, val_fmt='{0[0]}-{0[1]}') == '+--------+--------+--------+--------+--------+\n|        | col[0] | col[1] | col[2] | col[3] |\n+========+========+========+========+========+\n| row[0] | 0-11   | 1-10   | 2-9    | 3-8    |\n| row[1] | 4-7    | 5-6    | 6-5    | 7-4    |\n| row[2] | 8-3    | 9-2    | 10-1   | 11-0   |\n+--------+--------+--------+--------+--------+'
    formats = dict(down='!', along='_', thick_long='~', cross='%', title_heading='#')
    assert rst_table(cell_values, title='A title', format_chars=formats) == '#######\nA title\n#######\n\n%________%________%________%________%________%\n!        ! col[0] ! col[1] ! col[2] ! col[3] !\n%~~~~~~~~%~~~~~~~~%~~~~~~~~%~~~~~~~~%~~~~~~~~%\n! row[0] !  0.00  !  1.00  !  2.00  !  3.00  !\n! row[1] !  4.00  !  5.00  !  6.00  !  7.00  !\n! row[2] !  8.00  !  9.00  ! 10.00  ! 11.00  !\n%________%________%________%________%________%'
    formats['funny_value'] = '!'
    with pytest.raises(ValueError):
        rst_table(cell_values, title='A title', format_chars=formats)