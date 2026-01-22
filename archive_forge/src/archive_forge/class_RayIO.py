from modin.core.io import BaseIO
class RayIO(BaseIO):
    """Base class for doing I/O operations over Ray."""

    @classmethod
    def from_ray(cls, ray_obj):
        """
        Create a Modin `query_compiler` from a Ray Dataset.

        Parameters
        ----------
        ray_obj : ray.data.Dataset
            The Ray Dataset to convert from.

        Returns
        -------
        BaseQueryCompiler
            QueryCompiler containing data from the Ray Dataset.

        Notes
        -----
        This function must be implemented in every subclass
        otherwise NotImplementedError will be raised.
        """
        raise NotImplementedError(f"Modin dataset can't be created from `ray.data.Dataset` using {cls}.")

    @classmethod
    def to_ray(cls, modin_obj):
        """
        Convert a Modin DataFrame/Series to a Ray Dataset.

        Parameters
        ----------
        modin_obj : modin.pandas.DataFrame, modin.pandas.Series
            The Modin DataFrame/Series to convert.

        Returns
        -------
        ray.data.Dataset
            Converted object with type depending on input.

        Notes
        -----
        This function must be implemented in every subclass
        otherwise NotImplementedError will be raised.
        """
        raise NotImplementedError(f"`ray.data.Dataset` can't be created from Modin DataFrame/Series using {cls}.")