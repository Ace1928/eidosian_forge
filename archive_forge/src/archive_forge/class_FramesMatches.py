from collections import defaultdict
import numpy as np
from moviepy.decorators import use_clip_fps_by_default
class FramesMatches(list):

    def __init__(self, lst):
        list.__init__(self, sorted(lst, key=lambda e: e.d_max))

    def best(self, n=1, percent=None):
        if percent is not None:
            n = len(self) * percent / 100
        return self[0] if n == 1 else FramesMatches(self[:n])

    def filter(self, cond):
        """
        Returns a FramesMatches object obtained by filtering out the FramesMatch
        which do not satistify the condition ``cond``. ``cond`` is a function
        (FrameMatch -> bool).

        Examples
        ---------
        >>> # Only keep the matches corresponding to (> 1 second) sequences.
        >>> new_matches = matches.filter( lambda match: match.time_span > 1)
        """
        return FramesMatches(filter(cond, self))

    def save(self, filename):
        np.savetxt(filename, np.array([np.array(list(e)) for e in self]), fmt='%.03f', delimiter='\t')

    @staticmethod
    def load(filename):
        """ Loads a FramesMatches object from a file.
        >>> matching_frames = FramesMatches.load("somefile")
        """
        arr = np.loadtxt(filename)
        mfs = [FramesMatch(*e) for e in arr]
        return FramesMatches(mfs)

    @staticmethod
    def from_clip(clip, dist_thr, max_d, fps=None):
        """ Finds all the frames tht look alike in a clip, for instance to make a
        looping gif.

        This teturns a  FramesMatches object of the all pairs of frames with
        (t2-t1 < max_d) and whose distance is under dist_thr.

        This is well optimized routine and quite fast.

        Examples
        ---------
        
        We find all matching frames in a given video and turn the best match with
        a duration of 1.5s or more into a GIF:

        >>> from moviepy.editor import VideoFileClip
        >>> from moviepy.video.tools.cuts import find_matching_frames
        >>> clip = VideoFileClip("foo.mp4").resize(width=200)
        >>> matches = find_matching_frames(clip, 10, 3) # will take time
        >>> best = matches.filter(lambda m: m.time_span > 1.5).best()
        >>> clip.subclip(best.t1, best.t2).write_gif("foo.gif")

        Parameters
        -----------

        clip
          A MoviePy video clip, possibly transformed/resized
        
        dist_thr
          Distance above which a match is rejected
        
        max_d
          Maximal duration (in seconds) between two matching frames
        
        fps
          Frames per second (default will be clip.fps)
        
        """
        N_pixels = clip.w * clip.h * 3
        dot_product = lambda F1, F2: (F1 * F2).sum() / N_pixels
        F = {}

        def distance(t1, t2):
            uv = dot_product(F[t1]['frame'], F[t2]['frame'])
            u, v = (F[t1]['|F|sq'], F[t2]['|F|sq'])
            return np.sqrt(u + v - 2 * uv)
        matching_frames = []
        for t, frame in clip.iter_frames(with_times=True, logger='bar'):
            flat_frame = 1.0 * frame.flatten()
            F_norm_sq = dot_product(flat_frame, flat_frame)
            F_norm = np.sqrt(F_norm_sq)
            for t2 in list(F.keys()):
                if t - t2 > max_d:
                    F.pop(t2)
                else:
                    F[t2][t] = {'min': abs(F[t2]['|F|'] - F_norm), 'max': F[t2]['|F|'] + F_norm}
                    F[t2][t]['rejected'] = F[t2][t]['min'] > dist_thr
            t_F = sorted(F.keys())
            F[t] = {'frame': flat_frame, '|F|sq': F_norm_sq, '|F|': F_norm}
            for i, t2 in enumerate(t_F):
                if F[t2][t]['rejected']:
                    continue
                dist = distance(t, t2)
                F[t2][t]['min'] = F[t2][t]['max'] = dist
                F[t2][t]['rejected'] = dist >= dist_thr
                for t3 in t_F[i + 1:]:
                    t3t, t2t3 = (F[t3][t], F[t2][t3])
                    t3t['max'] = min(t3t['max'], dist + t2t3['max'])
                    t3t['min'] = max(t3t['min'], dist - t2t3['max'], t2t3['min'] - dist)
                    if t3t['min'] > dist_thr:
                        t3t['rejected'] = True
            matching_frames += [(t1, t, F[t1][t]['min'], F[t1][t]['max']) for t1 in F if t1 != t and (not F[t1][t]['rejected'])]
        return FramesMatches([FramesMatch(*e) for e in matching_frames])

    def select_scenes(self, match_thr, min_time_span, nomatch_thr=None, time_distance=0):
        """

        match_thr
          The smaller, the better-looping the gifs are.

        min_time_span
          Only GIFs with a duration longer than min_time_span (in seconds)
          will be extracted.

        nomatch_thr
          If None, then it is chosen equal to match_thr

        """
        if nomatch_thr is None:
            nomatch_thr = match_thr
        dict_starts = defaultdict(lambda: [])
        for start, end, d_min, d_max in self:
            dict_starts[start].append([end, d_min, d_max])
        starts_ends = sorted(dict_starts.items(), key=lambda k: k[0])
        result = []
        min_start = 0
        for start, ends_distances in starts_ends:
            if start < min_start:
                continue
            ends = [end for end, d_min, d_max in ends_distances]
            great_matches = [(end, d_min, d_max) for end, d_min, d_max in ends_distances if d_max < match_thr]
            great_long_matches = [(end, d_min, d_max) for end, d_min, d_max in great_matches if end - start > min_time_span]
            if not great_long_matches:
                continue
            poor_matches = {end for end, d_min, d_max in ends_distances if d_min > nomatch_thr}
            short_matches = {end for end in ends if end - start <= 0.6}
            if not poor_matches.intersection(short_matches):
                continue
            end = max((end for end, d_min, d_max in great_long_matches))
            end, d_min, d_max = next((e for e in great_long_matches if e[0] == end))
            result.append(FramesMatch(start, end, d_min, d_max))
            min_start = start + time_distance
        return FramesMatches(result)

    def write_gifs(self, clip, gif_dir):
        for start, end, _, _ in self:
            name = '%s/%08d_%08d.gif' % (gif_dir, 100 * start, 100 * end)
            clip.subclip(start, end).write_gif(name, verbose=False)