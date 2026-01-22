import abc
class DataIngester(metaclass=abc.ABCMeta):
    """Link between a data source and a data provider.

    A data ingester starts a reload operation in the background and
    provides a data provider as a view.
    """

    @property
    @abc.abstractmethod
    def data_provider(self):
        """Returns a `DataProvider`.

        It may be an error to dereference this before `start` is called.
        """
        pass

    @abc.abstractmethod
    def start(self):
        """Starts ingesting data.

        This may start a background thread or process, and will return
        once communication with that task is established. It won't block
        forever as data is reloaded.

        Must only be called once.
        """
        pass