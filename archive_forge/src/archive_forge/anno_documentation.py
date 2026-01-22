import enum
import gast
Recursively copies annotations in an AST tree.

  Args:
    node: ast.AST
    copy_map: Dict[Hashable, Hashable], maps a source anno key to a destination
        key. All annotations with the source key will be copied to identical
        annotations with the destination key.
    field_name: str
  