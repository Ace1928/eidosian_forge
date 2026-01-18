import pytest
from trio import run
from trio.lowlevel import RunVar, RunVarToken
from ... import _core
def test_runvar_sync() -> None:
    t1 = RunVar[str]('test1')

    async def sync_check() -> None:

        async def task1() -> None:
            t1.set('plaice')
            assert t1.get() == 'plaice'

        async def task2(tok: RunVarToken[str]) -> None:
            t1.reset(tok)
            with pytest.raises(LookupError):
                t1.get()
            t1.set('haddock')
        async with _core.open_nursery() as n:
            token = t1.set('cod')
            assert t1.get() == 'cod'
            n.start_soon(task1)
            await _core.wait_all_tasks_blocked()
            assert t1.get() == 'plaice'
            n.start_soon(task2, token)
            await _core.wait_all_tasks_blocked()
            assert t1.get() == 'haddock'
    run(sync_check)