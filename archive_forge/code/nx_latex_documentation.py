import numbers
import os
import networkx as nx
Write the latex code to draw the graph(s) onto `path`.

    This convenience function creates the latex drawing code as a string
    and writes that to a file ready to be compiled when `as_document` is True
    or ready to be ``import`` ed or ``include`` ed into your main LaTeX document.

    The `path` argument can be a string filename or a file handle to write to.

    Parameters
    ----------
    Gbunch : NetworkX graph or iterable of NetworkX graphs
        If Gbunch is a graph, it is drawn in a figure environment.
        If Gbunch is an iterable of graphs, each is drawn in a subfigure
        environment within a single figure environment.
    path : filename
        Filename or file handle to write to
    options : dict
        By default, TikZ is used with options: (others are ignored)::

            pos : string or dict or list
                The name of the node attribute on `G` that holds the position of each node.
                Positions can be sequences of length 2 with numbers for (x,y) coordinates.
                They can also be strings to denote positions in TikZ style, such as (x, y)
                or (angle:radius).
                If a dict, it should be keyed by node to a position.
                If an empty dict, a circular layout is computed by TikZ.
                If you are drawing many graphs in subfigures, use a list of position dicts.
            tikz_options : string
                The tikzpicture options description defining the options for the picture.
                Often large scale options like `[scale=2]`.
            default_node_options : string
                The draw options for a path of nodes. Individual node options override these.
            node_options : string or dict
                The name of the node attribute on `G` that holds the options for each node.
                Or a dict keyed by node to a string holding the options for that node.
            node_label : string or dict
                The name of the node attribute on `G` that holds the node label (text)
                displayed for each node. If the attribute is "" or not present, the node
                itself is drawn as a string. LaTeX processing such as ``"$A_1$"`` is allowed.
                Or a dict keyed by node to a string holding the label for that node.
            default_edge_options : string
                The options for the scope drawing all edges. The default is "[-]" for
                undirected graphs and "[->]" for directed graphs.
            edge_options : string or dict
                The name of the edge attribute on `G` that holds the options for each edge.
                If the edge is a self-loop and ``"loop" not in edge_options`` the option
                "loop," is added to the options for the self-loop edge. Hence you can
                use "[loop above]" explicitly, but the default is "[loop]".
                Or a dict keyed by edge to a string holding the options for that edge.
            edge_label : string or dict
                The name of the edge attribute on `G` that holds the edge label (text)
                displayed for each edge. If the attribute is "" or not present, no edge
                label is drawn.
                Or a dict keyed by edge to a string holding the label for that edge.
            edge_label_options : string or dict
                The name of the edge attribute on `G` that holds the label options for
                each edge. For example, "[sloped,above,blue]". The default is no options.
                Or a dict keyed by edge to a string holding the label options for that edge.
            caption : string
                The caption string for the figure environment
            latex_label : string
                The latex label used for the figure for easy referral from the main text
            sub_captions : list of strings
                The sub_caption string for each subfigure in the figure
            sub_latex_labels : list of strings
                The latex label for each subfigure in the figure
            n_rows : int
                The number of rows of subfigures to arrange for multiple graphs
            as_document : bool
                Whether to wrap the latex code in a document environment for compiling
            document_wrapper : formatted text string with variable ``content``.
                This text is called to evaluate the content embedded in a document
                environment with a preamble setting up the TikZ syntax.
            figure_wrapper : formatted text string
                This text is evaluated with variables ``content``, ``caption`` and ``label``.
                It wraps the content and if a caption is provided, adds the latex code for
                that caption, and if a label is provided, adds the latex code for a label.
            subfigure_wrapper : formatted text string
                This text evaluate variables ``size``, ``content``, ``caption`` and ``label``.
                It wraps the content and if a caption is provided, adds the latex code for
                that caption, and if a label is provided, adds the latex code for a label.
                The size is the vertical size of each row of subfigures as a fraction.

    See Also
    ========
    to_latex
    