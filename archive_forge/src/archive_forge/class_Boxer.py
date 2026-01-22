import operator
import os
import re
import subprocess
import tempfile
from functools import reduce
from optparse import OptionParser
from nltk.internals import find_binary
from nltk.sem.drt import (
from nltk.sem.logic import (
class Boxer:
    """
    This class is an interface to Johan Bos's program Boxer, a wide-coverage
    semantic parser that produces Discourse Representation Structures (DRSs).
    """

    def __init__(self, boxer_drs_interpreter=None, elimeq=False, bin_dir=None, verbose=False, resolve=True):
        """
        :param boxer_drs_interpreter: A class that converts from the
            ``AbstractBoxerDrs`` object hierarchy to a different object.  The
            default is ``NltkDrtBoxerDrsInterpreter``, which converts to the NLTK
            DRT hierarchy.
        :param elimeq: When set to true, Boxer removes all equalities from the
            DRSs and discourse referents standing in the equality relation are
            unified, but only if this can be done in a meaning-preserving manner.
        :param resolve: When set to true, Boxer will resolve all anaphoric DRSs and perform merge-reduction.
            Resolution follows Van der Sandt's theory of binding and accommodation.
        """
        if boxer_drs_interpreter is None:
            boxer_drs_interpreter = NltkDrtBoxerDrsInterpreter()
        self._boxer_drs_interpreter = boxer_drs_interpreter
        self._resolve = resolve
        self._elimeq = elimeq
        self.set_bin_dir(bin_dir, verbose)

    def set_bin_dir(self, bin_dir, verbose=False):
        self._candc_bin = self._find_binary('candc', bin_dir, verbose)
        self._candc_models_path = os.path.normpath(os.path.join(self._candc_bin[:-5], '../models'))
        self._boxer_bin = self._find_binary('boxer', bin_dir, verbose)

    def interpret(self, input, discourse_id=None, question=False, verbose=False):
        """
        Use Boxer to give a first order representation.

        :param input: str Input sentence to parse
        :param occur_index: bool Should predicates be occurrence indexed?
        :param discourse_id: str An identifier to be inserted to each occurrence-indexed predicate.
        :return: ``drt.DrtExpression``
        """
        discourse_ids = [discourse_id] if discourse_id is not None else None
        d, = self.interpret_multi_sents([[input]], discourse_ids, question, verbose)
        if not d:
            raise Exception(f'Unable to interpret: "{input}"')
        return d

    def interpret_multi(self, input, discourse_id=None, question=False, verbose=False):
        """
        Use Boxer to give a first order representation.

        :param input: list of str Input sentences to parse as a single discourse
        :param occur_index: bool Should predicates be occurrence indexed?
        :param discourse_id: str An identifier to be inserted to each occurrence-indexed predicate.
        :return: ``drt.DrtExpression``
        """
        discourse_ids = [discourse_id] if discourse_id is not None else None
        d, = self.interpret_multi_sents([input], discourse_ids, question, verbose)
        if not d:
            raise Exception(f'Unable to interpret: "{input}"')
        return d

    def interpret_sents(self, inputs, discourse_ids=None, question=False, verbose=False):
        """
        Use Boxer to give a first order representation.

        :param inputs: list of str Input sentences to parse as individual discourses
        :param occur_index: bool Should predicates be occurrence indexed?
        :param discourse_ids: list of str Identifiers to be inserted to each occurrence-indexed predicate.
        :return: list of ``drt.DrtExpression``
        """
        return self.interpret_multi_sents([[input] for input in inputs], discourse_ids, question, verbose)

    def interpret_multi_sents(self, inputs, discourse_ids=None, question=False, verbose=False):
        """
        Use Boxer to give a first order representation.

        :param inputs: list of list of str Input discourses to parse
        :param occur_index: bool Should predicates be occurrence indexed?
        :param discourse_ids: list of str Identifiers to be inserted to each occurrence-indexed predicate.
        :return: ``drt.DrtExpression``
        """
        if discourse_ids is not None:
            assert len(inputs) == len(discourse_ids)
            assert reduce(operator.and_, (id is not None for id in discourse_ids))
            use_disc_id = True
        else:
            discourse_ids = list(map(str, range(len(inputs))))
            use_disc_id = False
        candc_out = self._call_candc(inputs, discourse_ids, question, verbose=verbose)
        boxer_out = self._call_boxer(candc_out, verbose=verbose)
        drs_dict = self._parse_to_drs_dict(boxer_out, use_disc_id)
        return [drs_dict.get(id, None) for id in discourse_ids]

    def _call_candc(self, inputs, discourse_ids, question, verbose=False):
        """
        Call the ``candc`` binary with the given input.

        :param inputs: list of list of str Input discourses to parse
        :param discourse_ids: list of str Identifiers to be inserted to each occurrence-indexed predicate.
        :param filename: str A filename for the output file
        :return: stdout
        """
        args = ['--models', os.path.join(self._candc_models_path, ['boxer', 'questions'][question]), '--candc-printer', 'boxer']
        return self._call('\n'.join(sum(([f"<META>'{id}'"] + d for d, id in zip(inputs, discourse_ids)), [])), self._candc_bin, args, verbose)

    def _call_boxer(self, candc_out, verbose=False):
        """
        Call the ``boxer`` binary with the given input.

        :param candc_out: str output from C&C parser
        :return: stdout
        """
        f = None
        try:
            fd, temp_filename = tempfile.mkstemp(prefix='boxer-', suffix='.in', text=True)
            f = os.fdopen(fd, 'w')
            f.write(candc_out.decode('utf-8'))
        finally:
            if f:
                f.close()
        args = ['--box', 'false', '--semantics', 'drs', '--resolve', ['false', 'true'][self._resolve], '--elimeq', ['false', 'true'][self._elimeq], '--format', 'prolog', '--instantiate', 'true', '--input', temp_filename]
        stdout = self._call(None, self._boxer_bin, args, verbose)
        os.remove(temp_filename)
        return stdout

    def _find_binary(self, name, bin_dir, verbose=False):
        return find_binary(name, path_to_bin=bin_dir, env_vars=['CANDC'], url='http://svn.ask.it.usyd.edu.au/trac/candc/', binary_names=[name, name + '.exe'], verbose=verbose)

    def _call(self, input_str, binary, args=[], verbose=False):
        """
        Call the binary with the given input.

        :param input_str: A string whose contents are used as stdin.
        :param binary: The location of the binary to call
        :param args: A list of command-line arguments.
        :return: stdout
        """
        if verbose:
            print('Calling:', binary)
            print('Args:', args)
            print('Input:', input_str)
            print('Command:', binary + ' ' + ' '.join(args))
        if input_str is None:
            cmd = [binary] + args
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        else:
            cmd = 'echo "{}" | {} {}'.format(input_str, binary, ' '.join(args))
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        stdout, stderr = p.communicate()
        if verbose:
            print('Return code:', p.returncode)
            if stdout:
                print('stdout:\n', stdout, '\n')
            if stderr:
                print('stderr:\n', stderr, '\n')
        if p.returncode != 0:
            raise Exception('ERROR CALLING: {} {}\nReturncode: {}\n{}'.format(binary, ' '.join(args), p.returncode, stderr))
        return stdout

    def _parse_to_drs_dict(self, boxer_out, use_disc_id):
        lines = boxer_out.decode('utf-8').split('\n')
        drs_dict = {}
        i = 0
        while i < len(lines):
            line = lines[i]
            if line.startswith('id('):
                comma_idx = line.index(',')
                discourse_id = line[3:comma_idx]
                if discourse_id[0] == "'" and discourse_id[-1] == "'":
                    discourse_id = discourse_id[1:-1]
                drs_id = line[comma_idx + 1:line.index(')')]
                i += 1
                line = lines[i]
                assert line.startswith(f'sem({drs_id},')
                if line[-4:] == "').'":
                    line = line[:-4] + ').'
                assert line.endswith(').'), f"can't parse line: {line}"
                search_start = len(f'sem({drs_id},[')
                brace_count = 1
                drs_start = -1
                for j, c in enumerate(line[search_start:]):
                    if c == '[':
                        brace_count += 1
                    if c == ']':
                        brace_count -= 1
                        if brace_count == 0:
                            drs_start = search_start + j + 1
                            if line[drs_start:drs_start + 3] == "','":
                                drs_start = drs_start + 3
                            else:
                                drs_start = drs_start + 1
                            break
                assert drs_start > -1
                drs_input = line[drs_start:-2].strip()
                parsed = self._parse_drs(drs_input, discourse_id, use_disc_id)
                drs_dict[discourse_id] = self._boxer_drs_interpreter.interpret(parsed)
            i += 1
        return drs_dict

    def _parse_drs(self, drs_string, discourse_id, use_disc_id):
        return BoxerOutputDrsParser([None, discourse_id][use_disc_id]).parse(drs_string)