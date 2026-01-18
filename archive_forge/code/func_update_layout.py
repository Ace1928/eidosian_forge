from plotly.basedatatypes import BaseFigure
def update_layout(self, dict1=None, overwrite=False, **kwargs) -> 'Figure':
    """

        Update the properties of the figure's layout with a dict and/or with
        keyword arguments.

        This recursively updates the structure of the original
        layout with the values in the input dict / keyword arguments.

        Parameters
        ----------
        dict1 : dict
            Dictionary of properties to be updated
        overwrite: bool
            If True, overwrite existing properties. If False, apply updates
            to existing properties recursively, preserving existing
            properties that are not specified in the update operation.
        kwargs :
            Keyword/value pair of properties to be updated

        Returns
        -------
        BaseFigure
            The Figure object that the update_layout method was called on

        """
    return super(Figure, self).update_layout(dict1, overwrite, **kwargs)