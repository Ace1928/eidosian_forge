from typing_extensions import override
from pytorch_lightning.profilers.profiler import Profiler
class PassThroughProfiler(Profiler):
    """This class should be used when you don't want the (small) overhead of profiling.

    The Trainer uses this class by default.

    """

    @override
    def start(self, action_name: str) -> None:
        pass

    @override
    def stop(self, action_name: str) -> None:
        pass