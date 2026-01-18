from pythran.passmanager import NodeAnalysis

    Count the number of nodes included in a node

    This has nothing to do with execution time or whatever,
    its mainly use is to prevent the AST from growing too much when unrolling

    >>> import gast as ast
    >>> from pythran import passmanager, backend
    >>> node = ast.parse("if 1: return 3")
    >>> pm = passmanager.PassManager("test")
    >>> print(pm.gather(NodeCount, node))
    5
    