class UnsetMetadataPassedError(ValueError):
    """Exception class to raise if a metadata is passed which is not explicitly         requested (metadata=True) or not requested (metadata=False).

    .. versionadded:: 1.3

    Parameters
    ----------
    message : str
        The message

    unrequested_params : dict
        A dictionary of parameters and their values which are provided but not
        requested.

    routed_params : dict
        A dictionary of routed parameters.
    """

    def __init__(self, *, message, unrequested_params, routed_params):
        super().__init__(message)
        self.unrequested_params = unrequested_params
        self.routed_params = routed_params