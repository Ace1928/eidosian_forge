class InputRejected(Exception):
    """Input rejected by ast transformer.

    Raise this in your NodeTransformer to indicate that InteractiveShell should
    not execute the supplied input.
    """