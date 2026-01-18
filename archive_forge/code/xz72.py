from ai_logic import expectimax, simulate_move
from gui_utils import update_gui


@StandardDecorator()
def main_game_loop():
    board = initialize_game()
    score = 0
    game_over = False

    while not game_over:
        _, best_move = expectimax(board, depth=3, playerTurn=True)
        board, move_score = simulate_move(board, best_move)
        score += move_score
        update_gui(board, score)
        game_over = is_game_over(board)

        if not game_over:
            add_random_tile(board)
            update_gui(board, score)
