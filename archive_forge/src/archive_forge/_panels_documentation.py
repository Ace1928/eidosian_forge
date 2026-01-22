from mplfinance._helpers import _list_of_dict
from mplfinance._arg_validators import _valid_panel_id
import pandas as pd

    Create and return a DataFrame containing panel information
    and Axes objects for each panel, etc.

    We allow up to 32 panels, identified by their panel id (panid)
    which is an integer 0 through 31.  

    Parameters
    ----------
    figure       : pyplot.Figure
        figure on which to create the Axes for the panels

    config       : dict
        config dict from `mplfinance.plot()`
        
    Config
    ------
    The following items are used from `config`:

    num_panels   : integer (0-31) or None
        number of panels to create

    addplot      : dict or None
        value for the `addplot=` kwarg passed into `mplfinance.plot()`

    volume_panel : integer (0-31) or None
        panel id (0-number_of_panels)

    main_panel   : integer (0-31) or None
        panel id (0-number_of_panels)

    panel_ratios : sequence or None
        sequence of relative sizes for the panels;

        NOTE: If len(panel_ratios) == number of panels (regardless
        of whether number of panels was specified or inferred),
        then panel ratios are the relative sizes of each panel,
        in panel id order, 0 through N (where N = number of panels).

        If len(panel_ratios) != number of panels, then len(panel_ratios)
        must equal 2, and panel_ratios[0] is the relative size for the 'main'
        panel, and panel_ratios[1] is the relative size for all other panels.

        If the number of panels == 1, the panel_ratios is ignored.

    
Returns
    ----------
    panels  : pandas.DataFrame
        dataframe indexed by panel id (panid) and having the following columns:
          axes           : tuple of matplotlib.Axes (primary and secondary) for each column.
          used secondary : bool indicating whether or not the seconday Axes is in use.
          relative size  : height of panel as proportion of sum of all relative sizes

    