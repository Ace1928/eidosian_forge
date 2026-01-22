class CommittedButStillApplying(Timeout):
    """The write or transaction was committed, but some entities or index rows
  may not have been fully updated. Those updates should automatically be
  applied soon. You can roll them forward immediately by reading one of the
  entities inside a transaction.
  """