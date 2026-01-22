class MultipleProjectNames(Error):
    """Configuration file had both "application:" and "project:" fields.

  A configuration file can specify the project name using either the old-style
  "application: name" syntax or the newer "project: name" syntax, but not both.
  """