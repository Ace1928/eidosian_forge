import os
from apache_beam.io.filesystems import FileSystems
from apache_beam.pipeline import Pipeline
from .logging import get_logger
class BeamPipeline(Pipeline):
    """Wrapper over `apache_beam.pipeline.Pipeline` for convenience"""

    def is_local(self):
        runner = self._options.get_all_options().get('runner')
        return runner in [None, 'DirectRunner', 'PortableRunner']