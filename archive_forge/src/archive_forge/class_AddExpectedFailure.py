from twisted.protocols.amp import Boolean, Command, Integer, Unicode
class AddExpectedFailure(Command):
    """
    Add an expected failure.
    """
    arguments = [(b'testName', NativeString()), (b'errorStreamId', Integer()), (b'todo', NativeString())]
    response = [(b'success', Boolean())]