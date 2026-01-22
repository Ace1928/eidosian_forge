import pytest
import textwrap
from numpy.testing import assert_array_equal, assert_equal, assert_raises
import numpy as np
from numpy.f2py.tests import util
class TestMiscCharacter(util.F2PyTest):
    suffix = '.f90'
    fprefix = 'test_misc_character'
    code = textwrap.dedent(f"""\n       subroutine {fprefix}_gh18684(x, y, m)\n         character(len=5), dimension(m), intent(in) :: x\n         character*5, dimension(m), intent(out) :: y\n         integer i, m\n         !f2py integer, intent(hide), depend(x) :: m = f2py_len(x)\n         do i=1,m\n           y(i) = x(i)\n         end do\n       end subroutine {fprefix}_gh18684\n\n       subroutine {fprefix}_gh6308(x, i)\n         integer i\n         !f2py check(i>=0 && i<12) i\n         character*5 name, x\n         common name(12)\n         name(i + 1) = x\n       end subroutine {fprefix}_gh6308\n\n       subroutine {fprefix}_gh4519(x)\n         character(len=*), intent(in) :: x(:)\n         !f2py intent(out) x\n         integer :: i\n         ! Uncomment for debug printing:\n         !do i=1, size(x)\n         !   print*, "x(",i,")=", x(i)\n         !end do\n       end subroutine {fprefix}_gh4519\n\n       pure function {fprefix}_gh3425(x) result (y)\n         character(len=*), intent(in) :: x\n         character(len=len(x)) :: y\n         integer :: i\n         do i = 1, len(x)\n           j = iachar(x(i:i))\n           if (j>=iachar("a") .and. j<=iachar("z") ) then\n             y(i:i) = achar(j-32)\n           else\n             y(i:i) = x(i:i)\n           endif\n         end do\n       end function {fprefix}_gh3425\n\n       subroutine {fprefix}_character_bc_new(x, y, z)\n         character, intent(in) :: x\n         character, intent(out) :: y\n         !f2py character, depend(x) :: y = x\n         !f2py character, dimension((x=='a'?1:2)), depend(x), intent(out) :: z\n         character, dimension(*) :: z\n         !f2py character, optional, check(x == 'a' || x == 'b') :: x = 'a'\n         !f2py callstatement (*f2py_func)(&x, &y, z)\n         !f2py callprotoargument character*, character*, character*\n         if (y.eq.x) then\n           y = x\n         else\n           y = 'e'\n         endif\n         z(1) = 'c'\n       end subroutine {fprefix}_character_bc_new\n\n       subroutine {fprefix}_character_bc_old(x, y, z)\n         character, intent(in) :: x\n         character, intent(out) :: y\n         !f2py character, depend(x) :: y = x[0]\n         !f2py character, dimension((*x=='a'?1:2)), depend(x), intent(out) :: z\n         character, dimension(*) :: z\n         !f2py character, optional, check(*x == 'a' || x[0] == 'b') :: x = 'a'\n         !f2py callstatement (*f2py_func)(x, y, z)\n         !f2py callprotoargument char*, char*, char*\n          if (y.eq.x) then\n           y = x\n         else\n           y = 'e'\n         endif\n         z(1) = 'c'\n       end subroutine {fprefix}_character_bc_old\n    """)

    def test_gh18684(self):
        f = getattr(self.module, self.fprefix + '_gh18684')
        x = np.array(['abcde', 'fghij'], dtype='S5')
        y = f(x)
        assert_array_equal(x, y)

    def test_gh6308(self):
        f = getattr(self.module, self.fprefix + '_gh6308')
        assert_equal(self.module._BLNK_.name.dtype, np.dtype('S5'))
        assert_equal(len(self.module._BLNK_.name), 12)
        f('abcde', 0)
        assert_equal(self.module._BLNK_.name[0], b'abcde')
        f('12345', 5)
        assert_equal(self.module._BLNK_.name[5], b'12345')

    def test_gh4519(self):
        f = getattr(self.module, self.fprefix + '_gh4519')
        for x, expected in [('a', dict(shape=(), dtype=np.dtype('S1'))), ('text', dict(shape=(), dtype=np.dtype('S4'))), (np.array(['1', '2', '3'], dtype='S1'), dict(shape=(3,), dtype=np.dtype('S1'))), (['1', '2', '34'], dict(shape=(3,), dtype=np.dtype('S2'))), (['', ''], dict(shape=(2,), dtype=np.dtype('S1')))]:
            r = f(x)
            for k, v in expected.items():
                assert_equal(getattr(r, k), v)

    def test_gh3425(self):
        f = getattr(self.module, self.fprefix + '_gh3425')
        assert_equal(f('abC'), b'ABC')
        assert_equal(f(''), b'')
        assert_equal(f('abC12d'), b'ABC12D')

    @pytest.mark.parametrize('state', ['new', 'old'])
    def test_character_bc(self, state):
        f = getattr(self.module, self.fprefix + '_character_bc_' + state)
        c, a = f()
        assert_equal(c, b'a')
        assert_equal(len(a), 1)
        c, a = f(b'b')
        assert_equal(c, b'b')
        assert_equal(len(a), 2)
        assert_raises(Exception, lambda: f(b'c'))