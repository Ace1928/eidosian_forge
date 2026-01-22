from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleDevtoolsRemotebuildbotCommandEvents(_messages.Message):
    """CommandEvents contains counters for the number of warnings and errors
  that occurred during the execution of a command.

  Enums:
    CmUsageValueValuesEnum: Indicates if and how Container Manager is being
      used for task execution.
    OutputLocationValueValuesEnum: Indicates whether output files and/or
      output directories were found relative to the execution root or to the
      user provided work directory or both or none.
    UsedOverlayValueValuesEnum: Indicates whether overlay was used.

  Fields:
    cmUsage: Indicates if and how Container Manager is being used for task
      execution.
    dockerCacheHit: Indicates whether we are using a cached Docker image
      (true) or had to pull the Docker image (false) for this command.
    dockerImageName: Docker Image name.
    inputCacheMissBytes: The input cache miss rate as a fraction of the total
      size of input files.
    inputCacheMissFiles: The input cache miss rate as a fraction of the number
      of input files.
    numErrors: The number of errors reported.
    numWarnings: The number of warnings reported.
    outputLocation: Indicates whether output files and/or output directories
      were found relative to the execution root or to the user provided work
      directory or both or none.
    usedAsyncContainer: Indicates whether an asynchronous container was used
      for execution.
    usedOverlay: Indicates whether overlay was used.
  """

    class CmUsageValueValuesEnum(_messages.Enum):
        """Indicates if and how Container Manager is being used for task
    execution.

    Values:
      CONFIG_NONE: Container Manager is disabled or not running for this
        execution.
      CONFIG_MATCH: Container Manager is enabled and there was a matching
        container available for use during execution.
      CONFIG_MISMATCH: Container Manager is enabled, but there was no matching
        container available for execution.
    """
        CONFIG_NONE = 0
        CONFIG_MATCH = 1
        CONFIG_MISMATCH = 2

    class OutputLocationValueValuesEnum(_messages.Enum):
        """Indicates whether output files and/or output directories were found
    relative to the execution root or to the user provided work directory or
    both or none.

    Values:
      LOCATION_UNDEFINED: Location is set to LOCATION_UNDEFINED for tasks
        where the working directorty is not specified or is identical to the
        execution root directory.
      LOCATION_NONE: No output files or directories were found neither
        relative to the execution root directory nor relative to the working
        directory.
      LOCATION_EXEC_ROOT_RELATIVE: Output files or directories were found
        relative to the execution root directory but not relative to the
        working directory.
      LOCATION_WORKING_DIR_RELATIVE: Output files or directories were found
        relative to the working directory but not relative to the execution
        root directory.
      LOCATION_EXEC_ROOT_AND_WORKING_DIR_RELATIVE: Output files or directories
        were found both relative to the execution root directory and relative
        to the working directory.
      LOCATION_EXEC_ROOT_RELATIVE_OUTPUT_OUTSIDE_WORKING_DIR: Output files or
        directories were found relative to the execution root directory but
        not relative to the working directory. In addition at least one output
        file or directory was found outside of the working directory such that
        a working-directory-relative-path would have needed to start with a
        `..`.
      LOCATION_EXEC_ROOT_AND_WORKING_DIR_RELATIVE_OUTPUT_OUTSIDE_WORKING_DIR:
        Output files or directories were found both relative to the execution
        root directory and relative to the working directory. In addition at
        least one exec-root-relative output file or directory was found
        outside of the working directory such that a working-directory-
        relative-path would have needed to start with a `..`.
    """
        LOCATION_UNDEFINED = 0
        LOCATION_NONE = 1
        LOCATION_EXEC_ROOT_RELATIVE = 2
        LOCATION_WORKING_DIR_RELATIVE = 3
        LOCATION_EXEC_ROOT_AND_WORKING_DIR_RELATIVE = 4
        LOCATION_EXEC_ROOT_RELATIVE_OUTPUT_OUTSIDE_WORKING_DIR = 5
        LOCATION_EXEC_ROOT_AND_WORKING_DIR_RELATIVE_OUTPUT_OUTSIDE_WORKING_DIR = 6

    class UsedOverlayValueValuesEnum(_messages.Enum):
        """Indicates whether overlay was used.

    Values:
      OVERLAY_UNSPECIFIED: Indicates that whether or not overlay was used is
        unspecified. This applies, for example, to non-CommandTask actions.
      OVERLAY_ENABLED: Indicates that overlay was used for a CommandTask
        action.
      OVERLAY_DISABLED: Indicates that overlay was not used for a CommandTask
        action.
    """
        OVERLAY_UNSPECIFIED = 0
        OVERLAY_ENABLED = 1
        OVERLAY_DISABLED = 2
    cmUsage = _messages.EnumField('CmUsageValueValuesEnum', 1)
    dockerCacheHit = _messages.BooleanField(2)
    dockerImageName = _messages.StringField(3)
    inputCacheMissBytes = _messages.FloatField(4, variant=_messages.Variant.FLOAT)
    inputCacheMissFiles = _messages.FloatField(5, variant=_messages.Variant.FLOAT)
    numErrors = _messages.IntegerField(6, variant=_messages.Variant.UINT64)
    numWarnings = _messages.IntegerField(7, variant=_messages.Variant.UINT64)
    outputLocation = _messages.EnumField('OutputLocationValueValuesEnum', 8)
    usedAsyncContainer = _messages.BooleanField(9)
    usedOverlay = _messages.EnumField('UsedOverlayValueValuesEnum', 10)