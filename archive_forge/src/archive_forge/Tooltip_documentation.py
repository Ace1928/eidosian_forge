from dash.development.base_component import Component, _explicitize_args
A Tooltip component.
    A tooltip with an absolute position.

    Keyword arguments:

    - children (a list of or a singular dash component, string or number; optional):
        The contents of the tooltip.

    - id (string; optional):
        The ID of this component, used to identify dash components in
        callbacks. The ID needs to be unique across all of the components
        in an app.

    - background_color (string; default 'white'):
        Color of the tooltip background, as a CSS color string.

    - bbox (dict; optional):
        The bounding box coordinates of the item to label, in px relative
        to the positioning parent of the Tooltip component.

        `bbox` is a dict with keys:

        - x0 (number; optional)

        - x1 (number; optional)

        - y0 (number; optional)

        - y1 (number; optional)

    - border_color (string; default '#d6d6d6'):
        Color of the tooltip border, as a CSS color string.

    - className (string; default ''):
        The class of the tooltip.

    - direction (a value equal to: 'top', 'right', 'bottom', 'left'; default 'right'):
        The side of the `bbox` on which the tooltip should open.

    - loading_state (dict; optional):
        Object that holds the loading state object coming from
        dash-renderer.

        `loading_state` is a dict with keys:

        - component_name (string; optional):
            Holds the name of the component that is loading.

        - is_loading (boolean; optional):
            Determines if the component is loading or not.

        - prop_name (string; optional):
            Holds which property is loading.

    - loading_text (string; default 'Loading...'):
        The text displayed in the tooltip while loading.

    - show (boolean; default True):
        Whether to show the tooltip.

    - style (dict; optional):
        The style of the tooltip.

    - targetable (boolean; default False):
        Whether the tooltip itself can be targeted by pointer events. For
        tooltips triggered by hover events, typically this should be left
        `False` to avoid the tooltip interfering with those same events.

    - zindex (number; default 1):
        The `z-index` CSS property to assign to the tooltip. Components
        with higher values will be displayed on top of components with
        lower values.