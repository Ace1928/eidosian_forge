from sys import version_info
from pyflakes import messages as m
from pyflakes.test.harness import TestCase, skip, skipIf
class TestAsyncStatements(TestCase):

    def test_asyncDef(self):
        self.flakes('\n        async def bar():\n            return 42\n        ')

    def test_asyncDefAwait(self):
        self.flakes("\n        async def read_data(db):\n            await db.fetch('SELECT ...')\n        ")

    def test_asyncDefUndefined(self):
        self.flakes('\n        async def bar():\n            return foo()\n        ', m.UndefinedName)

    def test_asyncFor(self):
        self.flakes('\n        async def read_data(db):\n            output = []\n            async for row in db.cursor():\n                output.append(row)\n            return output\n        ')

    def test_asyncForUnderscoreLoopVar(self):
        self.flakes('\n        async def coro(it):\n            async for _ in it:\n                pass\n        ')

    def test_loopControlInAsyncFor(self):
        self.flakes("\n        async def read_data(db):\n            output = []\n            async for row in db.cursor():\n                if row[0] == 'skip':\n                    continue\n                output.append(row)\n            return output\n        ")
        self.flakes("\n        async def read_data(db):\n            output = []\n            async for row in db.cursor():\n                if row[0] == 'stop':\n                    break\n                output.append(row)\n            return output\n        ")

    def test_loopControlInAsyncForElse(self):
        self.flakes('\n        async def read_data(db):\n            output = []\n            async for row in db.cursor():\n                output.append(row)\n            else:\n                continue\n            return output\n        ', m.ContinueOutsideLoop)
        self.flakes('\n        async def read_data(db):\n            output = []\n            async for row in db.cursor():\n                output.append(row)\n            else:\n                break\n            return output\n        ', m.BreakOutsideLoop)

    def test_asyncWith(self):
        self.flakes('\n        async def commit(session, data):\n            async with session.transaction():\n                await session.update(data)\n        ')

    def test_asyncWithItem(self):
        self.flakes('\n        async def commit(session, data):\n            async with session.transaction() as trans:\n                await trans.begin()\n                ...\n                await trans.end()\n        ')

    def test_matmul(self):
        self.flakes('\n        def foo(a, b):\n            return a @ b\n        ')

    def test_formatstring(self):
        self.flakes("\n        hi = 'hi'\n        mom = 'mom'\n        f'{hi} {mom}'\n        ")

    def test_raise_notimplemented(self):
        self.flakes('\n        raise NotImplementedError("This is fine")\n        ')
        self.flakes('\n        raise NotImplementedError\n        ')
        self.flakes('\n        raise NotImplemented("This isn\'t gonna work")\n        ', m.RaiseNotImplemented)
        self.flakes('\n        raise NotImplemented\n        ', m.RaiseNotImplemented)