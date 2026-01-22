def apply_func_graph_transforms(func_graph):
    """Applies registered transformations to FuncGraph."""
    for transform in FUNC_GRAPH_TRANSFORMS:
        transform(func_graph)