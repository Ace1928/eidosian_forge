import numpy as np
import matplotlib.pyplot as plt
def test_stem_remove():
    ax = plt.gca()
    st = ax.stem([1, 2], [1, 2])
    st.remove()