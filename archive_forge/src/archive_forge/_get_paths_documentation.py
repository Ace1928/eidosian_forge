from ._utils import AttributeDict
from . import exceptions

    Return a path with `requests_pathname_prefix` and leading and trailing
    slashes stripped from it. Also, if None is passed in, None is returned.
    Use this function with `get_relative_path` in callbacks that deal
    with `dcc.Location` `pathname` routing.
    That is, your usage may look like:
    ```
    app.layout = html.Div([
        dcc.Location(id='url'),
        html.Div(id='content')
    ])
    @dash.callback(Output('content', 'children'), [Input('url', 'pathname')])
    def display_content(path):
        page_name = dash.strip_relative_path(path)
        if not page_name:  # None or ''
            return html.Div([
                dcc.Link(href=dash.get_relative_path('/page-1')),
                dcc.Link(href=dash.get_relative_path('/page-2')),
            ])
        elif page_name == 'page-1':
            return chapters.page_1
        if page_name == "page-2":
            return chapters.page_2
    ```
    Note that `chapters.page_1` will be served if the user visits `/page-1`
    _or_ `/page-1/` since `strip_relative_path` removes the trailing slash.

    Also note that `strip_relative_path` is compatible with
    `get_relative_path` in environments where `requests_pathname_prefix` set.
    In some deployment environments, like Dash Enterprise,
    `requests_pathname_prefix` is set to the application name, e.g. `my-dash-app`.
    When working locally, `requests_pathname_prefix` might be unset and
    so a relative URL like `/page-2` can just be `/page-2`.
    However, when the app is deployed to a URL like `/my-dash-app`, then
    `dash.get_relative_path('/page-2')` will return `/my-dash-app/page-2`

    The `pathname` property of `dcc.Location` will return '`/my-dash-app/page-2`'
    to the callback.
    In this case, `dash.strip_relative_path('/my-dash-app/page-2')`
    will return `'page-2'`

    For nested URLs, slashes are still included:
    `dash.strip_relative_path('/page-1/sub-page-1/')` will return
    `page-1/sub-page-1`
    ```
    