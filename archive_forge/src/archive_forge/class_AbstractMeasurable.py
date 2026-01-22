import abc
class AbstractMeasurable(object):
    """A measurable quantity that can be registered as a metric"""

    @abc.abstractmethod
    def measure(self, config, now):
        """
        Measure this quantity and return the result

        Arguments:
            config (MetricConfig): The configuration for this metric
            now (int): The POSIX time in milliseconds the measurement
                is being taken

        Returns:
            The measured value
        """
        raise NotImplementedError