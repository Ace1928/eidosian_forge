import os
from glob import glob
import numpy as np
from ... import logging
from ...utils.filemanip import ensure_list, simplify_list, split_filename
from ..base import (
from .base import SPMCommand, SPMCommandInputSpec, scans_for_fnames, ImageFileSPM
def _make_matlab_command(self, _):
    script = 'con_index = %d;\n' % self.inputs.contrast_index
    script += 'cluster_forming_thr = %f;\n' % self.inputs.height_threshold
    script += "stat_filename = '%s';\n" % self.inputs.stat_image
    script += 'extent_threshold = %d;\n' % self.inputs.extent_threshold
    script += "load '%s'\n" % self.inputs.spm_mat_file
    script += "\nFWHM  = SPM.xVol.FWHM;\ndf = [SPM.xCon(con_index).eidf SPM.xX.erdf];\nSTAT = SPM.xCon(con_index).STAT;\nR = SPM.xVol.R;\nS = SPM.xVol.S;\nn = 1;\n\nvoxelwise_P_Bonf = spm_P_Bonf(cluster_forming_thr,df,STAT,S,n)\nvoxelwise_P_RF = spm_P_RF(1,0,cluster_forming_thr,df,STAT,R,n)\n\nstat_map_vol = spm_vol(stat_filename);\n[stat_map_data, stat_map_XYZmm] = spm_read_vols(stat_map_vol);\n\nZ = stat_map_data(:);\nZum = Z;\n\n        switch STAT\n            case 'Z'\n                VPs = (1-spm_Ncdf(Zum)).^n;\n                voxelwise_P_uncor = (1-spm_Ncdf(cluster_forming_thr)).^n\n            case 'T'\n                VPs = (1 - spm_Tcdf(Zum,df(2))).^n;\n                voxelwise_P_uncor = (1 - spm_Tcdf(cluster_forming_thr,df(2))).^n\n            case 'X'\n                VPs = (1-spm_Xcdf(Zum,df(2))).^n;\n                voxelwise_P_uncor = (1-spm_Xcdf(cluster_forming_thr,df(2))).^n\n            case 'F'\n                VPs = (1 - spm_Fcdf(Zum,df)).^n;\n                voxelwise_P_uncor = (1 - spm_Fcdf(cluster_forming_thr,df)).^n\n        end\n        VPs = sort(VPs);\n\nvoxelwise_P_FDR = spm_P_FDR(cluster_forming_thr,df,STAT,n,VPs)\n\nV2R        = 1/prod(FWHM(stat_map_vol.dim > 1));\n\nclusterwise_P_RF = spm_P_RF(1,extent_threshold*V2R,cluster_forming_thr,df,STAT,R,n)\n\n[x,y,z] = ind2sub(size(stat_map_data),(1:numel(stat_map_data))');\nXYZ = cat(1, x', y', z');\n\n[u, CPs, ue] = spm_uc_clusterFDR(0.05,df,STAT,R,n,Z,XYZ,V2R,cluster_forming_thr);\n\nclusterwise_P_FDR = spm_P_clusterFDR(extent_threshold*V2R,df,STAT,R,n,cluster_forming_thr,CPs')\n"
    return script