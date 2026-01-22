from . import r_function
import pandas as pd
Simulate dataset with cellular backbone.

    The backbone determines the overall dynamic process during a simulation.
    It consists of a set of gene modules, which regulate each other such that
    expression of certain genes change over time in a specific manner.

    DyngenSimulate is a Python wrapper for the R package Dyngen.
    Default values obtained from Github vignettes.
    For more details, read about Dyngen on Github_.

    .. _Github: https://github.com/dynverse/dyngen

    Parameters
    ----------
    backbone: string
           Backbone name from dyngen list of backbones.
           Get list with get_backbones()).
    num_cells: int, optional (default: 500)
           Number of cells.
    num_tfs: int, optional (default: 100)
           Number of transcription factors.
           The TFs are the main drivers of the molecular changes in the simulation.
           A TF can only be regulated by other TFs or itself.

           NOTE: If num_tfs input is less than nrow(backbone$module_info),
           Dyngen will default to nrow(backbone$module_info).
           This quantity varies between backbones and with each run (without seed).
           It is generally less than 75.
           It is recommended to input num_tfs >= 100 to stabilize the output.
    num_targets: int, optional (default: 50)
           Number of target genes.
           Target genes are regulated by a TF or another target gene,
           but are always downstream of at least one TF.
    num_hks: int, optional (default: 25)
           Number of housekeeping genees.
           Housekeeping genes are completely separate from any TFs or target genes.
    simulation_census_interval: int, optional (default: 10)
           Stores the abundance levels only after a specific interval has passed.
           The lower the interval, the higher detail of simulation trajectory retained,
           though many timepoints will contain similar information.
    compute_cellwise_grn: boolean, optional (default: False)
           If True, computes the ground truth cellwise gene regulatory networks.
           Also outputs ground truth bulk (entire dataset) regulatory network.
           NOTE: Increases compute time significantly.
    compute_rna_velocity: boolean, optional (default: False)
           If true, computes the ground truth propensity ratios after simulation.
           NOTE: Increases compute time significantly.
    n_jobs: int, optional (default: 8)
           Number of cores to use.
    random_state: int, optional (default: None)
           Fixes seed for simulation generator.
    verbose: boolean, optional (default: True)
           Data generation verbosity.
    force_num_cells: boolean, optional (default: False)
           Dyngen occassionally produces fewer cells than specified.
           Set this flag to True to rerun Dyngen until correct cell count is reached.

    Returns
    -------
    Dictionary data of pd.DataFrames:
    data['cell_info']: pd.DataFrame, shape (n_cells, 4)
           Columns: cell_id, step_ix, simulation_i, sim_time
           sim_time is the simulated timepoint for a given cell.

    data['expression']: pd.DataFrame, shape (n_cells, n_genes)
           Log-transformed counts with dropout.

    If compute_cellwise_grn is True,
    data['bulk_grn']: pd.DataFrame, shape (n_tf_target_interactions, 4)
           Columns: regulator, target, strength, effect.
           Strength is positive and unbounded.
           Effect is either +1 (for activation) or -1 (for inhibition).

    data['cellwise_grn']: pd.DataFrame, shape (n_tf_target_interactions_per_cell, 4)
           Columns: cell_id, regulator, target, strength.
           The output does not include all edges per cell.
           The regulatory effect lies between [−1, 1], where -1 is complete inhibition
           of target by TF, +1 is maximal activation of target by TF,
           and 0 is inactivity of the regulatory interaction between R and T.

    If compute_rna_velocity is True,
    data['rna_velocity']: pd.DataFrame, shape (n_cells, n_genes)
           Propensity ratios for each cell.

    Example
    --------
    >>> import scprep
    >>> scprep.run.dyngen.install()
    >>> backbones = scprep.run.dyngen.get_backbones()
    >>> data = scprep.run.DyngenSimulate(backbone=backbones[0])
    