from twisted.protocols.amp import Boolean, Command, Integer, Unicode
class AddSuccess(Command):
    """
    Add a success.
    """
    arguments = [(b'testName', NativeString())]
    response = [(b'success', Boolean())]