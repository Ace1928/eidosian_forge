from twisted.protocols.amp import Boolean, Command, Integer, Unicode
class AddUnexpectedSuccess(Command):
    """
    Add an unexpected success.
    """
    arguments = [(b'testName', NativeString()), (b'todo', NativeString())]
    response = [(b'success', Boolean())]