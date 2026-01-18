import os
import subprocess
import shutil
import tempfile
from chempy import Reaction, ReactionSystem, Substance
from ..graph import rsys2dot, rsys2graph
from ..testing import requires, skipif
@requires('numpy')
@skipif(dot_missing, reason='graphviz not installed? (dot command missing)')
def test_rsys2dot():
    rsys = _get_rsys()
    assert list(map(str.strip, rsys2dot(rsys))) == ['digraph "None" {', '"A" [fontcolor=maroon label="A"];', '"B" [fontcolor=darkgreen label="B"];', '{', 'node [label="r1",shape=diamond]', 'r1', '}', '"A" -> "r1" [color=maroon,fontcolor=maroon,label="2"];', '"r1" -> "B" [color=darkgreen,fontcolor=darkgreen,label=""];', '}']