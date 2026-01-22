import sys
import os
import re
import copy
import warnings
import subprocess
import textwrap
from glob import glob
from functools import reduce
from configparser import NoOptionError
from configparser import RawConfigParser as ConfigParser
from distutils.errors import DistutilsError
from distutils.dist import Distribution
import sysconfig
from numpy.distutils import log
from distutils.util import get_platform
from numpy.distutils.exec_command import (
from numpy.distutils.misc_util import (is_sequence, is_string,
from numpy.distutils.command.config import config as cmd_config
from numpy.distutils import customized_ccompiler as _customized_ccompiler
from numpy.distutils import _shell_utils
import distutils.ccompiler
import tempfile
import shutil
import platform
class lapack_src_info(system_info):
    section = 'lapack_src'
    dir_env_var = 'LAPACK_SRC'
    notfounderror = LapackSrcNotFoundError

    def get_paths(self, section, key):
        pre_dirs = system_info.get_paths(self, section, key)
        dirs = []
        for d in pre_dirs:
            dirs.extend([d] + self.combine_paths(d, ['LAPACK*/SRC', 'SRC']))
        return [d for d in dirs if os.path.isdir(d)]

    def calc_info(self):
        src_dirs = self.get_src_dirs()
        src_dir = ''
        for d in src_dirs:
            if os.path.isfile(os.path.join(d, 'dgesv.f')):
                src_dir = d
                break
        if not src_dir:
            return
        allaux = '\n        ilaenv ieeeck lsame lsamen xerbla\n        iparmq\n        '
        laux = '\n        bdsdc bdsqr disna labad lacpy ladiv lae2 laebz laed0 laed1\n        laed2 laed3 laed4 laed5 laed6 laed7 laed8 laed9 laeda laev2\n        lagtf lagts lamch lamrg lanst lapy2 lapy3 larnv larrb larre\n        larrf lartg laruv las2 lascl lasd0 lasd1 lasd2 lasd3 lasd4\n        lasd5 lasd6 lasd7 lasd8 lasd9 lasda lasdq lasdt laset lasq1\n        lasq2 lasq3 lasq4 lasq5 lasq6 lasr lasrt lassq lasv2 pttrf\n        stebz stedc steqr sterf\n\n        larra larrc larrd larr larrk larrj larrr laneg laisnan isnan\n        lazq3 lazq4\n        '
        lasrc = '\n        gbbrd gbcon gbequ gbrfs gbsv gbsvx gbtf2 gbtrf gbtrs gebak\n        gebal gebd2 gebrd gecon geequ gees geesx geev geevx gegs gegv\n        gehd2 gehrd gelq2 gelqf gels gelsd gelss gelsx gelsy geql2\n        geqlf geqp3 geqpf geqr2 geqrf gerfs gerq2 gerqf gesc2 gesdd\n        gesv gesvd gesvx getc2 getf2 getrf getri getrs ggbak ggbal\n        gges ggesx ggev ggevx ggglm gghrd gglse ggqrf ggrqf ggsvd\n        ggsvp gtcon gtrfs gtsv gtsvx gttrf gttrs gtts2 hgeqz hsein\n        hseqr labrd lacon laein lags2 lagtm lahqr lahrd laic1 lals0\n        lalsa lalsd langb lange langt lanhs lansb lansp lansy lantb\n        lantp lantr lapll lapmt laqgb laqge laqp2 laqps laqsb laqsp\n        laqsy lar1v lar2v larf larfb larfg larft larfx largv larrv\n        lartv larz larzb larzt laswp lasyf latbs latdf latps latrd\n        latrs latrz latzm lauu2 lauum pbcon pbequ pbrfs pbstf pbsv\n        pbsvx pbtf2 pbtrf pbtrs pocon poequ porfs posv posvx potf2\n        potrf potri potrs ppcon ppequ pprfs ppsv ppsvx pptrf pptri\n        pptrs ptcon pteqr ptrfs ptsv ptsvx pttrs ptts2 spcon sprfs\n        spsv spsvx sptrf sptri sptrs stegr stein sycon syrfs sysv\n        sysvx sytf2 sytrf sytri sytrs tbcon tbrfs tbtrs tgevc tgex2\n        tgexc tgsen tgsja tgsna tgsy2 tgsyl tpcon tprfs tptri tptrs\n        trcon trevc trexc trrfs trsen trsna trsyl trti2 trtri trtrs\n        tzrqf tzrzf\n\n        lacn2 lahr2 stemr laqr0 laqr1 laqr2 laqr3 laqr4 laqr5\n        '
        sd_lasrc = '\n        laexc lag2 lagv2 laln2 lanv2 laqtr lasy2 opgtr opmtr org2l\n        org2r orgbr orghr orgl2 orglq orgql orgqr orgr2 orgrq orgtr\n        orm2l orm2r ormbr ormhr orml2 ormlq ormql ormqr ormr2 ormr3\n        ormrq ormrz ormtr rscl sbev sbevd sbevx sbgst sbgv sbgvd sbgvx\n        sbtrd spev spevd spevx spgst spgv spgvd spgvx sptrd stev stevd\n        stevr stevx syev syevd syevr syevx sygs2 sygst sygv sygvd\n        sygvx sytd2 sytrd\n        '
        cz_lasrc = '\n        bdsqr hbev hbevd hbevx hbgst hbgv hbgvd hbgvx hbtrd hecon heev\n        heevd heevr heevx hegs2 hegst hegv hegvd hegvx herfs hesv\n        hesvx hetd2 hetf2 hetrd hetrf hetri hetrs hpcon hpev hpevd\n        hpevx hpgst hpgv hpgvd hpgvx hprfs hpsv hpsvx hptrd hptrf\n        hptri hptrs lacgv lacp2 lacpy lacrm lacrt ladiv laed0 laed7\n        laed8 laesy laev2 lahef lanhb lanhe lanhp lanht laqhb laqhe\n        laqhp larcm larnv lartg lascl laset lasr lassq pttrf rot spmv\n        spr stedc steqr symv syr ung2l ung2r ungbr unghr ungl2 unglq\n        ungql ungqr ungr2 ungrq ungtr unm2l unm2r unmbr unmhr unml2\n        unmlq unmql unmqr unmr2 unmr3 unmrq unmrz unmtr upgtr upmtr\n        '
        sclaux = laux + ' econd '
        dzlaux = laux + ' secnd '
        slasrc = lasrc + sd_lasrc
        dlasrc = lasrc + sd_lasrc
        clasrc = lasrc + cz_lasrc + ' srot srscl '
        zlasrc = lasrc + cz_lasrc + ' drot drscl '
        oclasrc = ' icmax1 scsum1 '
        ozlasrc = ' izmax1 dzsum1 '
        sources = ['s%s.f' % f for f in (sclaux + slasrc).split()] + ['d%s.f' % f for f in (dzlaux + dlasrc).split()] + ['c%s.f' % f for f in clasrc.split()] + ['z%s.f' % f for f in zlasrc.split()] + ['%s.f' % f for f in (allaux + oclasrc + ozlasrc).split()]
        sources = [os.path.join(src_dir, f) for f in sources]
        src_dir2 = os.path.join(src_dir, '..', 'INSTALL')
        sources += [os.path.join(src_dir2, p + 'lamch.f') for p in 'sdcz']
        sources += [os.path.join(src_dir, p + 'larfp.f') for p in 'sdcz']
        sources += [os.path.join(src_dir, 'ila' + p + 'lr.f') for p in 'sdcz']
        sources += [os.path.join(src_dir, 'ila' + p + 'lc.f') for p in 'sdcz']
        sources = [f for f in sources if os.path.isfile(f)]
        info = {'sources': sources, 'language': 'f77'}
        self.set_info(**info)