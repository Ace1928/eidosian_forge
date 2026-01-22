from dash.development.base_component import Component, _explicitize_args
A Tab component.
    Part of dcc.Tabs - this is the child Tab component used to render a tabbed page.
    Its children will be set as the content of that tab, which if clicked will become visible.

    Keyword arguments:

    - children (a list of or a singular dash component, string or number; optional):
        The content of the tab - will only be displayed if this tab is
        selected.

    - id (string; optional):
        The ID of this component, used to identify dash components in
        callbacks. The ID needs to be unique across all of the components
        in an app.

    - className (string; optional):
        Appends a class to the Tab component.

    - disabled (boolean; default False):
        Determines if tab is disabled or not - defaults to False.

    - disabled_className (string; optional):
        Appends a class to the Tab component when it is disabled.

    - disabled_style (dict; default {    color: '#d6d6d6',}):
        Overrides the default (inline) styles when disabled.

    - label (string; optional):
        The tab's label.

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

    - selected_className (string; optional):
        Appends a class to the Tab component when it is selected.

    - selected_style (dict; optional):
        Overrides the default (inline) styles for the Tab component when
        it is selected.

    - style (dict; optional):
        Overrides the default (inline) styles for the Tab component.

    - value (string; optional):
        Value for determining which Tab is currently selected.