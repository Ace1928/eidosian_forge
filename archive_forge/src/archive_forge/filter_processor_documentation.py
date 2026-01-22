from .. import (
import stat
Convert a FileCopyCommand into a new FileCommand.

        :return: None if the copy is being ignored, otherwise a
          new FileCommand based on the whether the source and destination
          paths are inside or outside of the interesting locations.
          