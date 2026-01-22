from .series import SeriesDefault
class DateTimeDefault(SeriesDefault):
    """Builder for default-to-pandas methods which is executed under datetime accessor."""

    @classmethod
    def frame_wrapper(cls, df):
        """
        Get datetime accessor of the passed frame.

        Parameters
        ----------
        df : pandas.DataFrame

        Returns
        -------
        pandas.core.indexes.accessors.DatetimeProperties
        """
        return df.squeeze(axis=1).dt