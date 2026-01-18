from plotly.basedatatypes import BaseFigure
def update_smiths(self, patch=None, selector=None, overwrite=False, row=None, col=None, **kwargs) -> 'Figure':
    """
        Perform a property update operation on all smith objects
        that satisfy the specified selection criteria

        Parameters
        ----------
        patch: dict
            Dictionary of property updates to be applied to all
            smith objects that satisfy the selection criteria.
        selector: dict, function, or None (default None)
            Dict to use as selection criteria.
            smith objects will be selected if they contain
            properties corresponding to all of the dictionary's keys, with
            values that exactly match the supplied values. If None
            (the default), all smith objects are selected. If a
            function, it must be a function accepting a single argument and
            returning a boolean. The function will be called on each
            smith and those for which the function returned True will
            be in the selection.
        overwrite: bool
            If True, overwrite existing properties. If False, apply updates
            to existing properties recursively, preserving existing
            properties that are not specified in the update operation.
        row, col: int or None (default None)
            Subplot row and column index of smith objects to select.
            To select smith objects by row and column, the Figure
            must have been created using plotly.subplots.make_subplots.
            If None (the default), all smith objects are selected.
        **kwargs
            Additional property updates to apply to each selected
            smith object. If a property is specified in
            both patch and in **kwargs then the one in **kwargs
            takes precedence.
        Returns
        -------
        self
            Returns the Figure object that the method was called on
        """
    for obj in self.select_smiths(selector=selector, row=row, col=col):
        obj.update(patch, overwrite=overwrite, **kwargs)
    return self