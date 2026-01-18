from IPython.core.splitinput import LineInfo
from IPython.core.prefilter import AutocallChecker

    In [1]: def f1(a,b,c):
       ...:     return a+b+c
       ...:

    In [2]: def f2(a):
       ...:     return a + a
       ...:

    In [3]: def r(x):
       ...:     return True
       ...:

    In [4]: ;f2 a b c
    Out[4]: 'a b ca b c'

    In [5]: assert _ == "a b ca b c"

    In [6]: ,f1 a b c
    Out[6]: 'abc'

    In [7]: assert _ == 'abc'

    In [8]: print(_)
    abc

    In [9]: /f1 1,2,3
    Out[9]: 6

    In [10]: assert _ == 6

    In [11]: /f2 4
    Out[11]: 8

    In [12]: assert _ == 8

    In [12]: del f1, f2

    In [13]: ,r a
    Out[13]: True

    In [14]: assert _ == True

    In [15]: r'a'
    Out[15]: 'a'

    In [16]: assert _ == 'a'
    