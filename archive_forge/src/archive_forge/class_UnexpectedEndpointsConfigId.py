class UnexpectedEndpointsConfigId(Error):
    """Raised when an Endpoints config id is unexpected.

  An Endpoints config id is forbidden when the Endpoints rollout strategy is
  set to "managed".
  """