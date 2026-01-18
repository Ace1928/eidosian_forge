import pytest
from trio import run
from trio.lowlevel import RunVar, RunVarToken
from ... import _core
def test_runvar_smoketest() -> None:
    t1 = RunVar[str]('test1')
    t2 = RunVar[str]('test2', default='catfish')
    assert repr(t1) == "<RunVar name='test1'>"

    async def first_check() -> None:
        with pytest.raises(LookupError):
            t1.get()
        t1.set('swordfish')
        assert t1.get() == 'swordfish'
        assert t2.get() == 'catfish'
        assert t2.get(default='eel') == 'eel'
        t2.set('goldfish')
        assert t2.get() == 'goldfish'
        assert t2.get(default='tuna') == 'goldfish'

    async def second_check() -> None:
        with pytest.raises(LookupError):
            t1.get()
        assert t2.get() == 'catfish'
    run(first_check)
    run(second_check)