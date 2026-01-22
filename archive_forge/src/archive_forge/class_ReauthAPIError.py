class ReauthAPIError(ReauthError):
    """An exception for when reauth API returned something we can't handle."""

    def __init__(self, api_error):
        super(ReauthAPIError, self).__init__('Reauthentication challenge failed due to API error: {0}.'.format(api_error))