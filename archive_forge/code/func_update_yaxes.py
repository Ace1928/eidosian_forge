from plotly.basedatatypes import BaseFigure
def update_yaxes(self, patch=None, selector=None, overwrite=False, row=None, col=None, secondary_y=None, **kwargs) -> 'Figure':
    """
        Perform a property update operation on all yaxis objects
        that satisfy the specified selection criteria

        Parameters
        ----------
        patch: dict
            Dictionary of property updates to be applied to all
            yaxis objects that satisfy the selection criteria.
        selector: dict, function, or None (default None)
            Dict to use as selection criteria.
            yaxis objects will be selected if they contain
            properties corresponding to all of the dictionary's keys, with
            values that exactly match the supplied values. If None
            (the default), all yaxis objects are selected. If a
            function, it must be a function accepting a single argument and
            returning a boolean. The function will be called on each
            yaxis and those for which the function returned True will
            be in the selection.
        overwrite: bool
            If True, overwrite existing properties. If False, apply updates
            to existing properties recursively, preserving existing
            properties that are not specified in the update operation.
        row, col: int or None (default None)
            Subplot row and column index of yaxis objects to select.
            To select yaxis objects by row and column, the Figure
            must have been created using plotly.subplots.make_subplots.
            If None (the default), all yaxis objects are selected.
        secondary_y: boolean or None (default None)
            * If True, only select yaxis objects associated with the secondary
              y-axis of the subplot.
            * If False, only select yaxis objects associated with the primary
              y-axis of the subplot.
            * If None (the default), do not filter yaxis objects based on
              a secondary y-axis condition.

            To select yaxis objects by secondary y-axis, the Figure must
            have been created using plotly.subplots.make_subplots. See
            the docstring for the specs argument to make_subplots for more
            info on creating subplots with secondary y-axes.
        **kwargs
            Additional property updates to apply to each selected
            yaxis object. If a property is specified in
            both patch and in **kwargs then the one in **kwargs
            takes precedence.
        Returns
        -------
        self
            Returns the Figure object that the method was called on
        """
    for obj in self.select_yaxes(selector=selector, row=row, col=col, secondary_y=secondary_y):
        obj.update(patch, overwrite=overwrite, **kwargs)
    return self